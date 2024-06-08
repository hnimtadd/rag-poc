install:
	pip install -r requirements.txt

compose-up:
	docker-compose up --remove-orphans -d

compose-down:
	docker-compose down
