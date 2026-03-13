"""Compatibility entrypoint for deployments still pointing to traffic_data_app.py."""

from ryfast_app.app import main


if __name__ == "__main__":
    main()
