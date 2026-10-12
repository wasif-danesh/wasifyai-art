"""The hub's first tables: models and their versions.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE models (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            kind text NOT NULL CHECK (kind IN ('checkpoint', 'lora')),
            name text NOT NULL,
            slug text NOT NULL,
            tags text[] NOT NULL DEFAULT '{}',
            visibility text NOT NULL DEFAULT 'private',
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, product_id, slug)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE model_versions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            label text NOT NULL,
            family text,
            sha256 text,
            size_bytes bigint,
            trigger_words text[] NOT NULL DEFAULT '{}',
            recommended jsonb NOT NULL DEFAULT '{}',
            licence_note text,
            status text NOT NULL DEFAULT 'unconfirmed',
            source text NOT NULL DEFAULT 'scan',
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX model_versions_model_idx ON model_versions (model_id)")


def downgrade() -> None:
    op.execute("DROP TABLE model_versions")
    op.execute("DROP TABLE models")
