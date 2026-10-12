"""Files behind a version, and the suggestions read from them.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE model_versions
          ADD COLUMN family_hint text,
          ADD COLUMN trigger_hints text[] NOT NULL DEFAULT '{}'
        """
    )
    op.execute(
        """
        CREATE TABLE model_files (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            version_id uuid NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            rel_path text NOT NULL,
            folder text NOT NULL,
            size_bytes bigint NOT NULL,
            mtime double precision NOT NULL,
            header_sha256 text,
            tensor_count integer,
            meta jsonb NOT NULL DEFAULT '{}',
            status text NOT NULL DEFAULT 'present' CHECK (status IN ('present', 'missing')),
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, product_id, owner_id, rel_path)
        )
        """
    )
    op.execute("CREATE INDEX model_files_version_idx ON model_files (version_id)")


def downgrade() -> None:
    op.execute("DROP TABLE model_files")
    op.execute("ALTER TABLE model_versions DROP COLUMN trigger_hints, DROP COLUMN family_hint")
