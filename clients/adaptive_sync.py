"""
A federated learning client with support for Adaptive Synchronization Frequency.
"""

from config import Config
from clients import SimpleClient


class AdaptiveSyncClient(SimpleClient):
    """A federated learning client with support for Adaptive Synchronization
    Frequency.
    """
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'sync_frequency' in server_response:
            Config().trainer = Config().trainer._replace(
                epochs=server_response['sync_frequency'])
