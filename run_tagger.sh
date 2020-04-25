# tagger
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="data/" --bert_model="bert-base-uncased" --max_seq_length=128 --do_train --do_test --train_batch_size=32 --eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --do_lower_case --num_train_samples=10000 --task="SA" --output_dir="SA_tagger_output_bert_e3/"

CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="data/" --bert_model="bert-base-uncased" --max_seq_length=128 --do_train --do_test --train_batch_size=32 --eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2019 --do_lower_case --num_train_samples=10000 --task="NLI" --output_dir="NLI_tagger_output_bert_e3/"
