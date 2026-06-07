import binascii

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from core.config import get_settings

NONCE_SIZE = 12
TAG_SIZE = 16


class AESEncryption:
    """AES-256-GCM encryption for zero-trust payload storage."""

    def __init__(self, key_hex: str | None = None) -> None:
        settings = get_settings()
        key_material = key_hex or settings.AES_SECRET_KEY
        try:
            self._key = binascii.unhexlify(key_material)
        except binascii.Error as exc:
            raise ValueError("AES_SECRET_KEY must be a 64-character hex string (32 bytes).") from exc

        if len(self._key) != 32:
            raise ValueError("AES_SECRET_KEY must decode to exactly 32 bytes for AES-256.")

    def encrypt(self, data: str) -> bytes:
        nonce = get_random_bytes(NONCE_SIZE)
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data.encode("utf-8"))
        return nonce + tag + ciphertext

    def decrypt(self, blob: bytes) -> str:
        if len(blob) < NONCE_SIZE + TAG_SIZE + 1:
            raise ValueError("Encrypted payload is too short to decrypt.")

        nonce = blob[:NONCE_SIZE]
        tag = blob[NONCE_SIZE : NONCE_SIZE + TAG_SIZE]
        ciphertext = blob[NONCE_SIZE + TAG_SIZE :]
        cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode("utf-8")


def get_encryptor() -> AESEncryption:
    return AESEncryption()
