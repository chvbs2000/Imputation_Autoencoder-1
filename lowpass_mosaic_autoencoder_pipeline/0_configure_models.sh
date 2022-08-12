#/mnt/stsi/stsi0/raqueld/VMV_VCF_Extractions/chr22/

#bash configure_models.sh input.cfg out_root

full_models_per_node=40
#hp=1000_random_hyperparameters_500e.sh
train_script=dsae_lowpass.py
hp=best_pretrain_model.sh
#inference_script=inference_function.py
#accuracy_script=Compare_imputation_to_WGS.py
#inf_dependency1=configure_logging.py
#inf_dependency2=genotype_conversions.py
#plot_script=plot_evaluation_results_per_variant.R

if [ -z $1 ]; then
    echo "Please provide config file."
    echo "Usage: bash configure_models.sh input.cfg /path/to/output/root_dir"
    exit
fi


if [ -z $out_root ]; then
    out_root=$PWD
elif [ ! -d $out_root ]; then
    mkdir -p $out_root
    abs=$(readlink -e $2)
    out_root=$abs
fi

#out_root=$2

echo "Model training and validation directories will be at $out_root"

echo "#model directories" > $out_root/model_dir_list.txt

method=$(cat $1 | grep -v "#" | grep "^METHOD" | tr '=' ' ' | awk '{print $NF}')
traindir=$(cat $1 | grep -v "#" | grep "^TRAIN_DIR" | tr '=' ' ' | awk '{print $NF}')
#echo "train directory: $traindir"

wdir=$PWD
vram_script=$wdir/tools/estimate_VRAM_needed_for_autoencoder.py
#remove remaining job list from previous run
if [ -f ${out_root}/full_training_list.txt ]; then
    rm ${out_root}/full_training_list.txt*
fi

x=1

#touch ${out_root}/full_training_list.txt

echo $wdir

for i in $traindir/*.VMV1.gz; do
    
    #check if vmv is splited into segment:
    mul_seg=$(echo $i  | sed -e "s/\.VMV1.gz//g" | awk -F'.' '{print $NF}')
    if [ $mul_seg = "vcf" ]; then
        region=$(echo $i | sed -e 's/.*\.haplotypes\.//' | sed -e 's/.*_//g' | sed -e 's/\..*//g')
    else
        region=$(echo $i | sed -e "s/\.VMV1.gz//g" | awk -F'.' '{print $NF}' | cut -d'_' -f2)
    fi 
    
    #echo "$region"
    chr=$(echo $i | sed -e 's/.*chr//g' | sed -e 's/\..*//g')
    echo -e "${chr}_${region}\t$i"
    mdir="${out_root}/${chr}_${region}"
    echo "$mdir" >> $out_root/model_dir_list.txt
    if [ ! -d $mdir ]; then
        mkdir -p $mdir
    fi

    #echo "$mdir/train_best.sh" >> ${out_root}/full_training_list.txt

    VMV_name=$(basename $i)

    suffix=$(echo $VMV_name | sed -e 's/.*\.haplotypes//g')

    nvar=$(grep -v "#" $i | wc -l)

    # nmask=$((nvar-5))
    # mrate=$(bc -l <<< "$nmask/$nvar" | sed -e 's/^\./0\./g')

    hp_name=$(basename $hp)
    
    cp $hp $mdir/
    cp $train_script $mdir/

    cd $mdir

    
    echo "$nvar" > NVAR
    input=$i
    echo "$input" > INPUT

    mem_per_proc=$(python3 $vram_script $nvar | tail -n 1 | awk '{print $NF}' | sed -e 's/MiB//g')
    echo -e "$mem_per_proc" > VRAM

    if [ $method = "RANDOM" ]; then

        bashname="${VMV_name}_${hp_name}_file"
        #bashname="${VMV_name}"
        # while read line; do
        #     mi=$(echo $line | sed -e 's/.*--model_id //g' | sed -e 's/ .*//g')
        #     echo -e $line | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g" | sed -e "s~<my_input_file>~$input~g" | sed -e "s~<my_min_mask>~0.80~g" | sed -e "s~<my_max_mask>~$mrate~g" | sed -e "s~$~ 1\> $bashname\.$mi\.out 2\> $bashname\.$mi\.log~g"
        # done < $hp_name > $bashname
        
        #echo "i : $i"
        #segment=$(basename $i | cut -d'.' -f7)
        #echo "segment : ${segment}"
        
        while read line; do
            model_id=$x
            x=$(($x + 1))
            pretrain_model_subdir=$(basename ${i} | sed -e 's/\.gz//g')
            pretrain_model_dir=/gpfs/home/kchen/low_pass/best_model_weights/22_${region}/IMPUTATOR_${pretrain_model_subdir}
            #echo "get pretrained info ..."
            pretrain_model_pth=$(basename $(find ${pretrain_model_dir} -name "*pth"))
            pretrain_model_params=$(basename $(find ${pretrain_model_dir} -name "*py"))
            pretrain_model=$(echo ${pretrain_model_pth} | cut -d'_' -f1-2 | head -n 1)
            mi=$(echo pretrain_model_pth | cut -d'_' -f1-2)
            #echo ${pretrain_model}
            # hp_array=$(head -n 1 /gpfs/home/kchen/Imputation_Autoencoder/autoencoder_tuning_pipeline/vmv_profiling/best_hyperparameters_from_500000_xgb.sh | cut -d' ' -f12-41)
            hp_array=$(grep ${pretrain_model} /gpfs/home/kchen/Imputation_Autoencoder/autoencoder_tuning_pipeline/vmv_profiling/best_hyperparameters_from_500000_xgb.sh | head -n 1 | cut -d' ' -f12-41)
            
            echo -e $line | sed -e "s~<my_input_file>~${i}~g" | sed -e "s~<model_id>~model_lowpass_${model_id}~g" | sed -e "s~<pretrain_dir>~${pretrain_model_dir}~g" | sed -e "s~<pretrain_pth>~${pretrain_model_pth}~g" | sed -e "s~<pretrain_params>~${pretrain_model_params}~g" | sed -e "s~<hp>~${hp_array}~g" | sed -e "s~$~ 1\> $bashname\.$mi\.out 2\> $bashname\.$mi\.log~g"
            #echo "hpyerparams: ${hp_array}"
            #echo $line
            #echo ${hp_array}
            #echo -e $line | sed -e "s~<hp>~${hp_array}~g"
        done < $hp_name > ${bashname}
 
        
        #echo "CUDA_VISIBLE_DEVICES=<my_GPU_id> python3 dsae_lowpass.py --input ${i} --model_id model_lowpass_${model_id} --pretrain_model_dir ${pretrain_model_dir} --pretrain_pth ${pretrain_model_pth} --pretrain_params ${pretrain_model_params} ${hp_array}" >> ${out_root}/full_training_list.txt
        
        #split -l 100 -a 3 -d ${bashname} ${bashname}.

        #start at batch 0, or 1
        echo "$bashname.000" > BATCH_ID
                
    elif [ $method = "RAYTUNE" ]; then
        #insert RAYTUNE CONFIGURATION HERE
        echo -e "RAYTUNE COMMING SOON!!!"
        exit
    else
        echo -e "method not supported: $method. Please revise the value of METHOD in your config file $1"
        exit
    fi


    cd $wdir
    
done

cat ${out_root}/*/*file > ${out_root}/full_training_list.txt
split -l $full_models_per_node -a 3 -d ${out_root}/full_training_list.txt ${out_root}/full_training_list.txt



