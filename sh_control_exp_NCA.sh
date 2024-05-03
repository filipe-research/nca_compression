
#!/bin/bash
#inputs: $1-expname | $2-config_file
output_folder="Growing-Neural-Cellular-Automata/outputs/"
output_path=$output_folder$1"/ouput.ipynb"
conf_path="config/"$2


cd $output_folder
mkdir $1
cd ../..

#echo $output_path
#echo  $conf_path

#python control_exp_compressNCA.py --name $output_path --conf $conf_path
papermill NCA.ipynb  $output_path -f $conf_path --log-output  --progress-bar
