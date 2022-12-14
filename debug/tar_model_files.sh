
cd /home/mmajursk/Downloads/tmp/models

#ofp=/home/mmajursk/Downloads/tmp/models-packaged
#mkdir $ofp
#
#for i in $(seq 0 14)
#do
#	printf -v numStr "%07d" ${i}
#	echo "id-${numStr}x"
#
#	printf -v src "id-%07d*" ${i}
#	printf -v tgt "id-%07dx.tar.gz" ${i}
#
#  tar -czf $tgt $src
#  mv $tgt ${ofp}/
#done


for i in $(seq 0 14)
do
	printf -v numStr "%07d" ${i}
	echo "id-${numStr}x"

	printf -v src "id-%07d*" ${i}
	printf -v tgt "id-%07dx.tar.gz" ${i}

  tar -czf $tgt $src &
done

wait

