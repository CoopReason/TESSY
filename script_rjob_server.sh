

rjob submit --name=qwen3-8b-2 \
--gpu=8 \
--memory=122880 \
--cpu=16 \
--charged-group=llmkernel_gpu \
--private-machine=group \
--mount=gpfs://gpfs1/ailab-llmkernel:/mnt/shared-storage-user/ailab-llmkernel \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--image=registry.h.pjlab.org.cn/ailab-codex/gongjingyang-workspace:0908 \
--host-network=true \
-- bash /mnt/shared-storage-user/ailab-llmkernel/huangzixian/TESSY/start_server_qwen3_8b.sh


