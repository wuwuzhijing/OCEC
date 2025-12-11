 python demo_ocec.py --model deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
                     --ocec_model /103/guochuang/Code/myOCEC/runs/ocec_hq_finetune_progressive_v5.0/v1/ocec_best_epoch0249_f1_0.8607.onnx \
                     --video_dir /10/功能验证/001-功能验证视频-20240607/9_DSM/ \
                     --output_dir /10/cvz/guochuang/testdata/dsm/ocec_res/ \
                     --execution_provider cuda --inference_type fp16 -dwk