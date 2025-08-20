sandbox:
	docker build -t dualhrq:latest .
profile:
	docker run --rm dualhrq:latest