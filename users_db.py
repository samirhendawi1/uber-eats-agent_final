"""Simple user accounts for login/logout."""

USERS = {
    "samir": {
        "user_id": "user_samir",
        "username": "samir",
        "password": "password123",
        "full_name": "Samir Hendawi",
        "email": "samir@example.com",
        "phone": "+1 416-555-0101",
        "default_address": "123 Main St, Toronto, ON",
        "uber_one": True,
        "payment_methods": ["Visa ending in 4521", "Apple Pay", "Mastercard ending in 8834"],
    },
    "yuyan": {
        "user_id": "user_yuyan",
        "username": "yuyan",
        "password": "password123",
        "full_name": "Yuyan Zhang",
        "email": "yuyan@example.com",
        "phone": "+1 416-555-0202",
        "default_address": "200 Bay St, Toronto, ON",
        "uber_one": True,
        "payment_methods": ["Visa ending in 7712", "Apple Pay", "Mastercard ending in 3391"],
    },
    "jiaer": {
        "user_id": "user_jiaer",
        "username": "jiaer",
        "password": "password123",
        "full_name": "Jiaer Jiang",
        "email": "jiaer@example.com",
        "phone": "+1 416-555-0303",
        "default_address": "88 Harbour St, Toronto, ON",
        "uber_one": False,
        "payment_methods": ["Visa ending in 5543", "Apple Pay"],
    },
    "ce": {
        "user_id": "user_ce",
        "username": "ce",
        "password": "password123",
        "full_name": "Ce Shen",
        "email": "ce@example.com",
        "phone": "+1 416-555-0404",
        "default_address": "55 Bloor St W, Toronto, ON",
        "uber_one": False,
        "payment_methods": ["Visa ending in 9901", "Mastercard ending in 6688"],
    },
    "junyan": {
        "user_id": "user_junyan",
        "username": "junyan",
        "password": "password123",
        "full_name": "Junyan Yue",
        "email": "junyan@example.com",
        "phone": "+1 416-555-0505",
        "default_address": "10 Dundas St E, Toronto, ON",
        "uber_one": True,
        "payment_methods": ["Visa ending in 2234", "Apple Pay"],
    },
    "demo": {
        "user_id": "user_demo",
        "username": "demo",
        "password": "demo",
        "full_name": "Demo User",
        "email": "demo@example.com",
        "phone": "+1 416-555-9999",
        "default_address": "999 Demo Ave, Toronto, ON",
        "uber_one": False,
        "payment_methods": ["Visa ending in 0000"],
    },
}


def authenticate(username: str, password: str) -> dict | None:
    """Return user dict if credentials are valid, else None."""
    user = USERS.get(username.lower().strip())
    if user and user["password"] == password:
        return user
    return None


def get_user(username: str) -> dict | None:
    return USERS.get(username.lower().strip())
