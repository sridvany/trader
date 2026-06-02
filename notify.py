"""
notify.py — Telegram bildirim modülü.

Token ve chat ID ortam değişkeninden okunur (kodda saklanmaz):
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

GitHub Actions'ta bunlar Secrets olarak tanımlanır.
Lokal testte: export TELEGRAM_BOT_TOKEN=... ; export TELEGRAM_CHAT_ID=...

Token/chat tanımlı değilse mesaj sadece ekrana yazdırılır (bot yine çalışır).
"""
import os
import requests

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def send_telegram(msg):
    """Telegram'a mesaj gönderir. Başarı/başarısızlık bool döner."""
    if not TOKEN or not CHAT_ID:
        print("[TELEGRAM yok — env tanımsız]", msg)
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
        if r.status_code == 200:
            return True
        print(f"[TELEGRAM hata {r.status_code}] {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[TELEGRAM exception] {e}")
        return False


if __name__ == "__main__":
    # Bağlantı testi: python3 notify.py
    ok = send_telegram("✅ Test mesajı — bot bağlantısı çalışıyor.")
    print("Gönderim:", "BAŞARILI" if ok else "BAŞARISIZ (env değişkenlerini kontrol et)")
