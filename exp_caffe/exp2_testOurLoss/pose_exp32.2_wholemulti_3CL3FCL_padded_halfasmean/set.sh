parentFolder="/media/posenas1/shihenw/LEEDS_OC/h5file_multi_wholeImage_padded"
folderList=`ls $parentFolder`

#f=`ls $parentFolder | grep "_5_"`

lr=1e-4
#lr=( 0.01 0.001 0.0001 0.00001 )
fclLength=( 512 )

iteration=200000
snapshot=20000
stepsize=80000
visualbatch=8
batch_size=8
GPU=0

for fclLength_i in "${fclLength[@]}"
do
	echo $fclLength_i
	#part=`echo $f | awk -F "_" '{print $4}'`
	# echo $part
	#num=`echo $f | awk -F "_" '{print $2}'`
	# echo $num

	folderName="multi_FCL_${fclLength_i}_${lr}"
	echo $folderName

	if [ ! -d "$folderName" ]; then
		mkdir $folderName
		# Control will enter here if $DIRECTORY doesn't exist.
	fi

	# copy all needed files from pose_exp5
	source="../pose_exp32.1_wholemulti_3CL3FCL_padded_source"
	cp -R ${source}/matlab $folderName
	cp ${source}/pose_deploy.prototxt ${source}/pose_solver.prototxt ${source}/pose_train_test.prototxt ${source}/train_pose.sh $folderName

	# generate filelist_train and filelist_test
	numFiles=`ls $parentFolder | wc -l`
	trainFiles=264
	testFiles=24
	
	find $parentFolder -name "*" -type f | sort | head -n $trainFiles > $folderName/filelist_train.txt
	find $parentFolder -name "*" -type f | sort | tail -n $testFiles > $folderName/filelist_test.txt

	# run training
	cd $folderName
	#rm pose.log
	if [ ! -f pose.log ]; then
		echo "going to run in $folderName"
		sed -i "20s/.*/max_iter: ${iteration}/" pose_solver.prototxt
		sed -i "10s/.*/base_lr: ${lr}/" pose_solver.prototxt
		sed -i "22s/.*/snapshot: ${snapshot}/" pose_solver.prototxt
		sed -i "16s/.*/stepsize: ${stepsize}/" pose_solver.prototxt
		sed -i "9s/.*/    batch_size: ${batch_size}/" pose_train_test.prototxt
		sed -i "20s/.*/    batch_size: ${batch_size}/" pose_train_test.prototxt
		sh train_pose.sh $GPU > pose.log 2>&1
	fi
		sed -i "20s/.*/max_iter: ${iteration}/" pose_solver.prototxt
                sed -i "10s/.*/base_lr: ${lr}/" pose_solver.prototxt
                sed -i "22s/.*/snapshot: ${snapshot}/" pose_solver.prototxt
		sed -i "16s/.*/stepsize: ${stepsize}/" pose_solver.prototxt
		sed -i "9s/.*/    batch_size: ${batch_size}/" pose_train_test.prototxt
                sed -i "20s/.*/    batch_size: ${batch_size}/" pose_train_test.prototxt

		grep "loss" pose.log | grep "Iteration"	| awk '{print $6 " " $9}' | awk -F[,] '{print $1 " " $2}' > trainLoss.log
		grep "loss" pose.log | grep "Test"      | awk '{print $15}' > testLoss.log

		grep "loss" pose.log | grep "Iteration 0" | awk '{print $2}' >> ../summary
                grep "loss" pose.log | grep "Iteration ${iteration}" | awk '{print $2}' >> ../summary
	#fi
	cd ..
	echo "back to `pwd`....training done"

done
