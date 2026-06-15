 python demo_ocec.py --model deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
                     --ocec_model /103/guochuang/Code/myOCEC/runs/ocec_hq_finetune_progressive_v5.0/v1/ocec_best_epoch1005_f1_0.8909.onnx \
                     --video_dir /10/cv/gupengli/2025-11-11_2025-11-18_fatigue_videos/zhe_GC7372 \
                     --output_dir /10/cvz/guochuang/testdata/dsm/ocec_res/zhe_GC7372/ \
                     --execution_provider cuda --inference_type fp16 -dwk