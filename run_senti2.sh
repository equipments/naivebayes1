hadoop fs -rmr /tmp/natashac/sentiment/

hadoop jar /opt/cloudera/parcels/CDH-5.0.0-0.cdh5b2.p0.27/lib/hadoop-mapreduce/hadoop-streaming-2.2.0-cdh5.0.0-beta-2.jar \
           -input /user/hive/warehouse/mytest \
           -output /tmp/natashac/sentiment/ \
           -mapper "mapper_senti.py" \
           -reducer "reducer_senti5.py" \
           -file mapper_senti.py \
           -file reducer_senti5.py \
           -cmdenv dir1=http://ip-10-0-0-190.us-west-2.compute.internal:8088/home/ubuntu/aclImdb/train/pos \
           -cmdenv dir2=http://ip-10-0-0-190.us-west-2.compute.internal:8088/home/ubuntu/aclImdb/train/pos/ \

