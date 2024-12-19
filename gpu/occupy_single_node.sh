i=0
while [[ $i -lt $(nvidia-smi --list-gpus | wc -l) ]] ; do
   echo "occupying gpu $i"
   CUDA_VISIBLE_DEVICES=$i python3 /apdcephfs_cq10/share_1567347/share_info/jackjianli/projects_cq/taiji_client/run_anchun.py &
  (( i += 1 ))
done


# sh /apdcephfs_cq10/share_1567347/share_info/wendyzhang/gpu/occupy_single_node.sh