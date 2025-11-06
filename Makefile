
.PHONY: all train predict

train:
	@cd classifier && \
		python cli.py train --data-dir data/train --out-dir model/svm_v1

predict:
	@cd classifier && \
		python cli.py predict --image $(image) --model-dir model/svm_v1

classifier-worker:
	@cd classifier && \
		python worker.py


train-python:
	@cd classifier && \
		python cli.py train --data-dir data/train --out-dir model/svm_v1

predict-python:
	@cd classifier && \
		python cli.py predict --image $(image) --model-dir model/svm_v1

classifier-worker-python:
	@cd classifier && \
		python worker.py
