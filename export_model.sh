
python PaddleRS/deploy/export/export_model.py \
        --model_dir model/best_model/change_detection \
        --save_dir model/inference_model/change_detection \
        --fixed_input_shape [1024,1024]

read -p "按任意键继续！"