clear
python search_r1_like/local_dense_retriever/download.py   --save_path "$HOME/data/searchR1_processed_direct"
clear
bash ./search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh 
clear
bash ./search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh 
clear
python examples/data_preprocess/preprocess_search_r1_dataset.py --hf_repo_id PeterJinGo/nq_hotpotqa_train --local_dir /mnt/isdslab/data/searchR1_processed_direct
clear
bash ./search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh 
clear
clear
ls
exit
clear
bash ./search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh 
clear
bash ./search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh 
exit
