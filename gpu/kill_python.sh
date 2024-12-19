echo $NODE_IP_LIST > env.txt

#sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
sed "s/:.//g" env.txt | sed "s/,/\n/g" >  "pssh.hosts"
#yum install pssh
pssh -i -t 0 -h pssh.hosts pkill -9 python