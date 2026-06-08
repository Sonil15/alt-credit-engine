import pytest

from core.security import AESEncryption


def test_aes_round_trip():
    key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    encryptor = AESEncryption(key_hex=key)
    plaintext = '{"user_id": "test", "amount": 1234.56}'
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext


def test_aes_invalid_key():
    with pytest.raises(ValueError):
        AESEncryption(key_hex="not-a-valid-key")
