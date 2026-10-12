"""Jobs and creations.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE jobs (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            kind text NOT NULL DEFAULT 'generate',
            status text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
            request jsonb NOT NULL,
            progress real NOT NULL DEFAULT 0,
            error text,
            attempts integer NOT NULL DEFAULT 0,
            creation_id uuid,
            gpu_seconds real,
            created_at timestamptz NOT NULL DEFAULT now(),
            started_at timestamptz,
            finished_at timestamptz
        )
        """
    )
    op.execute(
        "CREATE INDEX jobs_owner_idx ON jobs (tenant_id, product_id, owner_id, created_at DESC)"
    )
    op.execute(
        """
        CREATE TABLE creations (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            kind text NOT NULL DEFAULT 'image',
            prompt text NOT NULL,
            params jsonb NOT NULL DEFAULT '{}',
            provider_id text NOT NULL,
            parent_id uuid REFERENCES creations(id) ON DELETE SET NULL,
            image_key text NOT NULL,
            thumb_key text NOT NULL,
            width integer NOT NULL,
            height integer NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX creations_owner_idx "
        "ON creations (tenant_id, product_id, owner_id, created_at DESC)"
    )
    op.execute(
        """
        CREATE TABLE creation_models (
            creation_id uuid NOT NULL REFERENCES creations(id) ON DELETE CASCADE,
            version_id uuid REFERENCES model_versions(id) ON DELETE SET NULL,
            role text NOT NULL DEFAULT 'base',
            weight real,
            name text NOT NULL,
            family text,
            rel_path text NOT NULL,
            header_sha256 text
        )
        """
    )
    op.execute("CREATE INDEX creation_models_creation_idx ON creation_models (creation_id)")


def downgrade() -> None:
    op.execute("DROP TABLE creation_models")
    op.execute("DROP TABLE creations")
    op.execute("DROP TABLE jobs")
