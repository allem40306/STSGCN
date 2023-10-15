generate(){
    # python3 main.py --config config/$1/individual_GLU_mask_emb.json --save > $1.txt
    python3 result.py --config config/$1/individual_GLU_mask_emb.json
    mv *$1* $2
}

experiment(){
    echo $1
    mkdir $1
    generate PEMS04 $1
    generate PEMS08 $1
    ls $1
}

experiment result01

# cd data
# python Temporal_Graph_gen.py --dataset Youbike --lag 6 --period 24
# cd ..