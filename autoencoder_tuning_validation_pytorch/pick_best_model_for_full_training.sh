
if [ -z $1 ] | [ -z $2 ] ; then
    echo
    echo "usage: bash pick_best_model_for_full_training.sh <overall_results_per_model> <training_script>"
    echo "    "
    echo "    <overall_results_per_model>: summary file generated by the plotting script (plot_evaluation_results_per_variant.R), more specifically: overall_results_per_model.tsv"
    echo "    <training_script>: the training script you generated from make_training_script_from_template.sh, before splitting by GPU"
    echo "    "
    echo "example: bash pick_best_model_for_full_training.sh models_1_154786497-155032195/plots/overall_results_per_model.tsv models_1_154786497-155032195/HRC.r1-1.EGA.GRCh37.chr1.haplotypes.154786474-156208223.m770.1_154786497-155032195.VMV1_100_random_hyperparameters_500e.sh"
    echo
    exit
fi


model_list=$(cat $2 | sed -e 's/.*--model_id //g' | sed -e 's/ .*//g')
pattern=$(echo $model_list | sed -e 's/ /\\\|/g')

best=$(cat $1 | grep -w "${pattern}" | sort -k2,2g | tail -n 1 | cut -f 1)

cat $2 | grep -w $best | sed -e 's/--max_epochs .*/--max_epochs 50000/g' | sed -e "s/$best/${best}_F/g" > ${2}.best

echo
echo -e "Best model is $best, new full training model name: ${best}_F"
echo -e "Full training script created at ${2}.best"
echo
