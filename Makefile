
.PHONY: all train predict

train:
	@docker compose stop classifier && \
	docker compose run --remove-orphans classifier python cli.py train --data-dir data/train --out-dir model/svm_v1 

predict:
	@docker compose stop classifier && \
	docker compose run --remove-orphans classifier python cli.py predict --image data/predict/$(image) --model-dir model/svm_v1 
