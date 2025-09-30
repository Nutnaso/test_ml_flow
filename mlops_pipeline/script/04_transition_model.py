# 04_transition_model.py — Transition model alias for Mushroom EfficientNet
from mlflow.tracking import MlflowClient


def transition_model_alias(model_name: str, alias: str, description: str | None = None):
    """
    Transition the latest version of a registered model to a given alias.

    Args:
        model_name (str): Name of the registered model in MLflow.
        alias (str): Alias to assign (e.g., 'champion', 'staging', 'production').
        description (str, optional): Description to update for the latest version.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise SystemExit(f"No versions found for model '{model_name}'.")

    # Select latest version
    latest_version = max(versions, key=lambda mv: int(mv.version))
    version_number = int(latest_version.version)
    print(f"ℹ️ Latest version for {model_name}: v{version_number}")

    # Try updating description, fallback to tag if fails
    if description:
        try:
            client.update_model_version(
                name=model_name,
                version=version_number,
                description=description
            )
        except Exception as e:
            print(f"[WARN] update_model_version failed, fallback to tag. Error: {e}")
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="description",
                value=description
            )

    # Set alias
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version_number
    )
    print(f"✅ Alias '{alias}' set on {model_name} v{version_number}.")


if __name__ == "__main__":
    # Example usage
    transition_model_alias(
        model_name="Mushroom-EfficientNet",
        alias="staging",
        description="EfficientNet trained on mushroom dataset v1"
    )
