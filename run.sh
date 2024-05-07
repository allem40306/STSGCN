experiment(){
    echo $1_$2
    mkdir $1_$2
    python3 main.py --config config/$1/individual_GLU_mask_emb.json --save --folder $1_$2 --test> $1_$2/$1.txt
}

experiment PEMS04 r02