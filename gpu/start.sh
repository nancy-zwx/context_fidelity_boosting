echo "------------------Installation Environment--------------------"

# python设置
export PYTHON=/usr/bin/python3.7
export PYTHON_HOME=/usr/lib/python3.7
export PATH=$PYTHON_HOME/bin:$PATH

# proxy设置
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

# NCCL设置
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=1

# HF设置
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
# export TRANSFORMERS_OFFLINE=1  # 如果模型已经下载过，强制使用本地缓存
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_ENDPOINT="https://huggingface.co"
export HF_TOKEN="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"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0

# 安装依赖

# pip install -r requirements.txt
pip install torch
pip install transformers
pip install --upgrade transformers
pip install accelerate
pip install evaluate
pip install rouge-score
pip install bert-score
pip install datasets
pip install tqdm
pip install numpy
pip install pandas
pip install scikit-learn
pip install ipdb
pip install sentencepiece
pip install nltk
pip install tensorboard
pip install time
pip install termcolor


echo "Verifying installations..."

i=0
while [[ $i -lt $(nvidia-smi --list-gpus | wc -l) ]] ; do
   echo "occupying gpu $i"
   CUDA_VISIBLE_DEVICES=$i python3 /apdcephfs_cq10/share_1567347/share_info/jackjianli/projects_cq/taiji_client/run_anchun.py &
  (( i += 1 ))
done
