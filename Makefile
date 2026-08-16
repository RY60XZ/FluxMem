-include .env

FLUXMEM_POSTGRES_DB ?= fluxmem
FLUXMEM_POSTGRES_USER ?= fluxmem
FLUXMEM_POSTGRES_PASSWORD ?= fluxmem_dev
FLUXMEM_POSTGRES_PORT ?= 5433
FLUXMEM_DATABASE_URL ?= postgresql+psycopg://$(FLUXMEM_POSTGRES_USER):$(FLUXMEM_POSTGRES_PASSWORD)@127.0.0.1:$(FLUXMEM_POSTGRES_PORT)/$(FLUXMEM_POSTGRES_DB)

export FLUXMEM_POSTGRES_DB
export FLUXMEM_POSTGRES_USER
export FLUXMEM_POSTGRES_PASSWORD
export FLUXMEM_POSTGRES_PORT

.PHONY: db-setup db-up db-migrate db-status db-shell db-logs db-stop db-down

db-setup: db-up db-migrate

db-up:
	docker compose up -d --wait postgres

db-migrate:
	FLUXMEM_DATABASE_URL="$(FLUXMEM_DATABASE_URL)" .venv/bin/alembic upgrade head

db-status:
	docker compose ps

db-shell:
	docker compose exec postgres psql -U "$(FLUXMEM_POSTGRES_USER)" -d "$(FLUXMEM_POSTGRES_DB)"

db-logs:
	docker compose logs --tail=100 postgres

db-stop:
	docker compose stop postgres

db-down:
	docker compose down
