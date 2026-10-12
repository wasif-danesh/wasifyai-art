"""LoRA weights must round-trip exactly (0.7 stays 0.7), so they are double precision.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.execute("ALTER TABLE creation_models ALTER COLUMN weight TYPE double precision")


def downgrade() -> None:
    op.execute("ALTER TABLE creation_models ALTER COLUMN weight TYPE real")
