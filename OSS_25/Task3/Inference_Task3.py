# -*- coding: utf-8 -*-

'''
File for Task3 inference
'''

# Import utils
from Utils_Inference import *

def inference(input_path, output_path, save_frames = False):
    print(f"Inference for Task3 using folder {input_path} for input and {output_path} for output.")
    ### List videos (currently supports only mp4)
    videos_for_inference = [video for video in listdir(input_path) if video.split('.')[-1] in ['mp4']]
    print(f"{len(videos_for_inference)} video{'s' if len(videos_for_inference) > 1 else ''} detected for inference")

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    #setup for SAM2
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

    ### SAM2
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device = device, apply_postprocessing = False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side= 64,
        points_per_batch= 64,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        )

    # Radiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(EXTRACTOR_PARAMS_FILE)

    # XGBoost 
    loaded = joblib.load(f"{BASE_PATH_CHECKPOINTS}/XGBoost/xgb_artifact.joblib")
    # adapt to whatever structure you stored; this expects a dict with keys like 'model','feature_names','target_names'
    model_xgb = loaded.get("model", loaded)  # if you saved model directly, loaded may already be the model
    model_feature_names = loaded.get("feature_names", None)
    target_names = loaded.get("target_names", None)
    
    if model_feature_names is None:
        raise ValueError("Model 'feature_names' not found in joblib artifact. Check what you saved in joblib.")
    ### UNET
    # Load models
    shape_pad = (256, 256) # TODO this can be other shape
    ALL_UNET_MODELS = {}    
    for key_name in CLASS_NAMES_INV.keys():
        features = [max(1, f ) for f in [64, 128, 256, 512]]  # because ch_div=2
        dropout = 0.25
        region_mask = key_name
        saved_path = f"{BASE_PATH_CHECKPOINTS}/UNET_weights/best_unet_chdiv_dropout_{dropout}_{region_mask}.pth"
    
        num_keypoints = num_keypoints_dict[region_mask]
    
        model = UNet(in_channels=3, out_channels=num_keypoints, features=features, dropout=0.0)
        model.load_state_dict(torch.load(saved_path, map_location="cpu"))
        model.cuda()
        model.eval()  # set to eval mode
        ALL_UNET_MODELS[key_name] = model
    print(f"Models loaded successfully!")
    
    for video in videos_for_inference:
        #extract frames from each video
        frames_list = {}
        sam2_masks = {}
        
        print(f"Opening video {input_path}/{video}")
        cap = cv2.VideoCapture(f'{input_path}/{video}') # Create a VideoCapture object and read from input file
         
        # Check if capture opened successfully
        if (cap.isOpened() == False): 
          print("Error opening video stream or file")
        # Get frame rate information # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = int(cap.get(5))
        # Get frame count # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frames_total = int(cap.get(7))
        
        # Read until video is completed
        for frame in range(frames_total): 
            # Capture frame-by-frame
            ret, img = cap.read()
            if not ret:# == False
                break
            if frame % fps == 0: # process at 1fps
                print(f"Extracting frame {frame}/{frames_total}")
                key = f"{video.split('.')[0]}_frame_{frame}"
                #get video name without .mp4 extension
                frames_list[key] = img
                #extract "segmentation" from sam2 output
                sam2_masks[key] = np.array([mask["segmentation"] for mask in mask_generator.generate(frames_list[key])]) 
                print(f"Detected {len(sam2_masks[key])} mask{'s' if len(sam2_masks[key]) > 1 else ''}")
        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        for idx_frame, key in enumerate(frames_list.keys()):
            #sabe the extracted frames if needed
            if (save_frames):
                cv2.imwrite(f"{output_path}/{key}.png", frames_list[key])
            input_list = []
            # --- Load data ---
            image_array = gray_img(frames_list[key])  # shape (H, W)
            H, W = image_array.shape
            masks_array = sam2_masks[key]
            mask_list = ensure_mask_stack_layout(masks_array, (H, W))
            print(f"Found {len(mask_list)} mask{'s' if len(mask_list) > 1 else ''}. Image shape: {H}x{W}")
            
            # Convert image and masks to SimpleITK and cast types
            image_sitk = sitk.GetImageFromArray(image_array)         # SimpleITK expects (z,y,x) convention
            image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)     # float image

            # input_list -> [ [image_sitk, mask_sitk], [], ...]
            for idx, mask_arr in enumerate(mask_list):
                mask_bin = (mask_arr > 0).astype(np.uint8)
                mask_sitk = sitk.GetImageFromArray(mask_bin)
                # get_biggest_object shouldn't be here because it doesn't work well for not continous objects
                #mask_sitk = get_biggest_object(mask_sitk)
                input_list.append([image_sitk, mask_sitk])

            logging.getLogger('radiomics').setLevel(logging.WARNING)
            results = run_parallel_extraction(input_list, extractor, max_workers=8)
                    
            ### XGBOOST 
            df, case_names = prepare_dataset(results, key) #was case_name
            
            # Check missing features and add zeros for them
            missing = [f for f in model_feature_names if f not in df.columns]
            if missing:
                print(f"Warning: Missing features in DataFrame; adding zero columns for: {missing}")
                for f in missing:
                    df[f] = 0.0
            
            # Reorder to model's expected order
            X_new = df[model_feature_names].values.astype('float32')

            # Predict
            y_pred = model_xgb.predict(X_new)
            y_proba = model_xgb.predict_proba(X_new) if hasattr(model_xgb, "predict_proba") else None
            if target_names is not None:
                pred_names = [target_names[int(p)] for p in y_pred]
            else:
                pred_names = [str(int(p)) for p in y_pred]
            
            predicted_masks = {}
            for mask_id, (case, probs) in enumerate(zip(case_names, y_proba)):
                best_idx = np.argmax(probs)               # index of highest probability
                best_name = target_names[best_idx]        # class name at that index
                best_prob = probs[best_idx]               # probability value
            
                # If it's a hand
                if best_name=='hand':
                    # Compute center point 
                    com_new = center_of_mass(mask= get_biggest_and_neighbors(mask_np=masks_array[mask_id].astype(np.uint8),
                                                                                min_size=10,
                                                                                distance_threshold=100)  
                                                                                ) # New mask
                    if 'left hand' not in predicted_masks or 'right hand' not in predicted_masks:
                        # if x coordinate is in the left side of the image, it is likely left hand
                        if com_new[1] < 1920//2:
                            predicted_masks['left hand'] = get_biggest_and_neighbors(mask_np=masks_array[mask_id].astype(np.uint8),
                                                                                min_size=10,
                                                                                distance_threshold=100)
                        # Else is likely right hand
                        else:
                            predicted_masks['right hand'] = get_biggest_and_neighbors(mask_np=masks_array[mask_id].astype(np.uint8),
                                                                                min_size=10,
                                                                                distance_threshold=100)               
                    # If the class hand already exists in the predicted_masks dict            
                    if best_name in predicted_masks: 
                        # check the distance
                        com_old = center_of_mass(mask=predicted_masks[best_name]) # old mask already saved in dict
                        p1, p2 = np.array(com_new), np.array(com_old)
                        distance = np.linalg.norm(p1 - p2)
                        
                        if distance>50:
                            if com_new[1]>com_old[1]:
                                # com_new is likely the right hand
                                predicted_masks['right hand'] = get_biggest_and_neighbors(mask_np=masks_array[mask_id].astype(np.uint8),
                                                                            min_size=10,
                                                                            distance_threshold=100
                                                                            ) #get_biggest_object Added to remove far away segmented pixels
                                predicted_masks['left hand'] = get_biggest_and_neighbors(mask_np=predicted_masks[best_name].astype(np.uint8),
                                                                            min_size=10,
                                                                            distance_threshold=100
                                                                            )
                            else:
                                predicted_masks['right hand'] = get_biggest_and_neighbors(mask_np=predicted_masks[best_name].astype(np.uint8),
                                                                            min_size=10,
                                                                            distance_threshold=100
                                                                            )
                                predicted_masks['left hand'] = get_biggest_and_neighbors(mask_np=masks_array[mask_id].astype(np.uint8),
                                                                            min_size=10,
                                                                            distance_threshold=100
                                                                            ) #get_biggest_object Added to remove far away segmented pixels

                predicted_masks[best_name] = masks_array[mask_id]
                print(f"Case: {case}, Pred: {best_name} (index={best_idx})")

            frame_n = int(key.split("_")[-1])
            ### Predict points
            inference_entries = [] 
            print(f"All objects detected: {predicted_masks.keys()}")
            for key_name in CLASS_NAMES_INV.keys():
                if (key_name.lower() in predicted_masks.keys()):
                    #initialize bounding box
                    bb_x, bb_y, bb_w, bb_h  = -1, -1, -1, -1
                    print("Predicting " + key_name)
                    image_array = frames_list[key]
                    image_array = np.array(image_array)
                    
                    mask_np = predicted_masks[key_name.lower()].astype(np.uint8)
                    bb_x, bb_y, bb_w, bb_h = get_bbox(mask_np)
                    
                    padded_case, crop_meta, pad_meta = generator(image_array, mask_np, shape_pad)
                    padded_case = padded_case.unsqueeze(0)  # -> (1, H, W)
            
                    pred_heatmap = ALL_UNET_MODELS[key_name](padded_case.cuda())
            
                    pred_coords = []
                    for heatmap in pred_heatmap[0].float().cpu().detach().numpy():
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        y /=shape_pad[1]
                        y -= 0.5
                        
                        x /=shape_pad[0]
                        x -= 0.5
                        pred_coords.append([x, y])  # store as (x, y)
                    pred_coords = np.array(pred_coords)  # shape (3, 2)
                    
                    pred_coords = np.array(pred_coords)  # shape (3, 2)
                    pred_coords = pred_coords[np.newaxis, ...] # Add a batch dim 
                    pred_coords = torch.from_numpy(pred_coords).float()  # shape: (1, 3, 2)
                    pred_adjusted_points = convert_points_back(pred_coords, crop_meta, pad_meta, shape_pad)

                    # Concatenate -1 for each point
                    neg_ones = torch.full((pred_adjusted_points.size(0), 1), -1)
                    pred_adjusted_points = torch.cat((pred_adjusted_points, neg_ones), dim=1)


                    
                    # id of the object ->  0	Left Hand  |  1 Right Hand  |  2 Scissors  | 3 Tweezers  |  4 Needle Holder  |  5 Needle
                    track_id = CLASS_NAMES_INV[key_name]
                    class_id = track_id # For some reason
            
                    entry = [
                        frame_n, track_id, class_id, bb_x, bb_y, bb_w, bb_h, *pred_adjusted_points
                    ]
                    clean_entry = flatten_entry(entry)
                    inference_entries.append(clean_entry)
                    print(clean_entry)
                    if(key_name == "Needle_Holder" and "Needle".lower() not in predicted_masks.keys()): 
                        print("Pretend we know the needle.")
                        #assume needle position is next to needle holder if not already detected
                        # id of the object ->  0	Left Hand  |  1 Right Hand  |  2 Scissors  | 3 Tweezers  |  4 Needle Holder  |  5 Needle
                        track_id = CLASS_NAMES_INV["Needle"]
                        class_id = track_id # For some reason
                        w, h = 20, 50 #aproximate size for needle

                        #check if needle holder is pointing left or right
                        if(pred_adjusted_points[0][0] < bb_x + ( bb_w / 2 )):
                            needle_bb_x = bb_x - w
                        else:
                            needle_bb_x = bb_x + bb_w
                        #check if needle holder is pointing up or down
                        if(pred_adjusted_points[0][1] > bb_y - ( bb_h / 2 )):
                            needle_bb_y = bb_y
                        else:
                            needle_bb_y = bb_y + bb_h - h
                        
                        needle_points = [ int(needle_bb_x+w/2), needle_bb_y+h, -1,          #center x top y
                                          int(needle_bb_x+w/2), int(needle_bb_y+h/2), -1,   #center x middle y
                                          int(needle_bb_x+w/2), needle_bb_y, -1             #center x bottom y
                                        ]
                        
                        entry = [
                            frame_n, track_id, class_id, needle_bb_x, needle_bb_y, w, h, *needle_points #assume bb for needle
                        ]
                        clean_entry = flatten_entry(entry)
                        inference_entries.append(clean_entry)
                        print(clean_entry)
                else:
                    print(f"{key_name} not predicted by SAM2 or detected by XGBoost.")
            #generate one output CSV per video
            with open(f"{output_path}/Task3_output_{video.split('.')[0]}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(inference_entries)  # writes each list as a row
                print(f"Saved in {output_path}/Task3_output_{video.split('.')[0]}.csv")
        
if __name__ == '__main__':
    inference("/input","/output") #test Container TF1