"""Training datasets, their images, and training runs.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE datasets (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            name text NOT NULL,
            trigger_word text NOT NULL,
            kind text NOT NULL DEFAULT 'image',
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX datasets_owner_idx ON datasets (tenant_id, product_id, owner_id)")
    op.execute(
        """
        CREATE TABLE dataset_images (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            key text NOT NULL,
            thumb_key text NOT NULL,
            width integer NOT NULL,
            height integer NOT NULL,
            caption text NOT NULL DEFAULT '',
            position integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX dataset_images_dataset_idx ON dataset_images (dataset_id, position)")
    op.execute(
        """
        CREATE TABLE training_runs (
            job_id uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
            dataset_id uuid REFERENCES datasets(id) ON DELETE SET NULL,
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            name text NOT NULL,
            trigger_word text NOT NULL,
            family text NOT NULL,
            preset text NOT NULL,
            params jsonb NOT NULL DEFAULT '{}',
            checkpoints jsonb NOT NULL DEFAULT '[]',
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX training_runs_owner_idx ON training_runs (tenant_id, product_id, owner_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE training_runs")
    op.execute("DROP TABLE dataset_images")
    op.execute("DROP TABLE datasets")
