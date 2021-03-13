#!/bin/bash
ifdir()
{
	for cmdd in "$@"
	do
		if  test -d "$cmdd";
		then
			echo the $cmdd directory exists 
		else
			mkdir $cmdd
		fi
	done 
}

# size/ (500 by 500 for example)
#	split/ (0.5 or 0.9)
#		target_size/ (0.5 to 1 kilohertz)
#			{model}_{n_obs}.txt ([exp, unif], [0.5 kh to 1 kh])
build_all="T" 
### currently in use:
if [ "$build_all" == "T" ];
then
	curr_directory="experiment_results";  ifdir "${curr_directory}"
	#for freq in "1k" "2k" "4k"
	#do
	for size in "small" "medium" "publish"
	do
		ifdir "${curr_directory}/${size}/"
			for split in "0.5" "0.9" "0.7"
			do
				ifdir "${curr_directory}/${size}/split_${split}"		
			done
	done
	#done

else
	ifdir "experiment_results" "experiment_results/0.5" "experiment_results/0.9"
	ifdir "experiment_results/0.5/medium" "experiment_results/0.9/medium"
	ifdir "experiment_results/0.5/medium/target_.5kh" "experiment_results/0.9/medium/target_1kh"
fi
tree "experiment_results"
# then you will, for this first experimental pass, create 
#four files: 
#	exp__1kh_obs.txt, unif__1kh_obs.txt 
#   and 
#	exp__.5kh_obs.txt, unif__0.5kh_obs.txt 
