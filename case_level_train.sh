pip -q install vit_pytorch linformer
pip -q install einops
pip -q install ipywidgets

python3 case_level_train.py --train_img $1 --jsonfile $2 --out_model cs_out.pth
python3 case_level_csv.py --model cs_out.pth --testdir $3 --out_csv $4