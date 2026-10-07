#!/usr/bin/env python3
"""
TÜBİTAK 2204-A: WebXR VR El Takibi ile Sürekli İşaret Dili Tanıma
Yerel Geliştirme ve Veri Toplama Sunucusu (HTTPS + REST API)

WebXR Hand Input API güvenlik gereği sadece HTTPS (veya localhost) üzerinden çalışır.
Meta Quest 2 gözlüğünden Wi-Fi üzerinden bağlanabilmek için bu sunucu SSL sertifikası
ile 8443 portunda çalışır ve VR'dan toplanan el verilerini anında kaydeder.
"""

import http.server
import ssl
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ai_pipeline" / "data"
SSL_DIR = BASE_DIR / "ssl"
PORT = 8443

DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_local_ip():
    """Bilgisayarın yerel ağdaki (Wi-Fi) IP adresini tespit eder."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class VRSignLanguageHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        # WebXR ve Cross-Origin gereksinimleri için CORS başlıkları
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_GET(self):
        # API: İstatistik endpoint'i
        if self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            stats = {}
            for json_file in DATA_DIR.glob("*.json"):
                parts = json_file.stem.split("_")
                label = parts[0] if parts else "unknown"
                stats[label] = stats.get(label, 0) + 1
            
            response = {
                "total_samples": sum(stats.values()),
                "classes": stats
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return
            
        return super().do_GET()

    def do_POST(self):
        # API: VR'dan gelen el takibi verisini doğrudan diske kaydetme
        if self.path == '/api/save_sample':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                label = data.get('label', 'unlabeled')
                timestamp = int(datetime.now().timestamp() * 1000)
                frames = data.get('frames', [])
                
                filename = f"{label}_{timestamp}.json"
                filepath = DATA_DIR / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"[VERİ KAYDEDİLDİ] Etiket: {label} | Kare Sayısı: {len(frames)} -> {filename}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    "status": "success",
                    "filename": filename,
                    "frame_count": len(frames),
                    "label": label
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                print(f"[HATA] Veri kaydedilemedi: {e}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

def run_server():
    cert_file = SSL_DIR / "cert.pem"
    key_file = SSL_DIR / "key.pem"
    
    if not cert_file.exists() or not key_file.exists():
        print(f"[HATA] SSL sertifikaları bulunamadı! Lütfen ssl/cert.pem ve ssl/key.pem dosyalarını kontrol edin.")
        sys.exit(1)

    local_ip = get_local_ip()
    server_address = ('0.0.0.0', PORT)
    httpd = http.server.HTTPServer(server_address, VRSignLanguageHandler)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print("=" * 65)
    print(" TÜBİTAK 2204-A WebXR VR İşaret Dili Sunucusu Başlatıldı ")
    print("=" * 65)
    print(f" Bilgisayardan test için:")
    print(f"   -> Veri Toplayıcı : https://localhost:{PORT}/webxr_collector/")
    print(f"   -> VR Çıkarım     : https://localhost:{PORT}/webxr_inference/")
    print("-" * 65)
    print(f" Meta Quest 2 Gözlüğünden Bağlanmak İçin (Aynı Wi-Fi ağında):")
    print(f"   -> Veri Toplayıcı : https://{local_ip}:{PORT}/webxr_collector/")
    print(f"   -> VR Çıkarım     : https://{local_ip}:{PORT}/webxr_inference/")
    print("-" * 65)
    print(" NOT (Quest 2 İpucu):")
    print("   Meta Quest Browser ilk girişte 'Bağlantınız gizli değil'")
    print("   uyarısı verirse 'Gelişmiş' (Advanced) -> 'Devam Et' (Proceed)")
    print("   seçeneğine tıklayarak yerel SSL sertifikasını onaylayın.")
    print("=" * 65)
    print("Sunucu dinleniyor (Durdurmak için Ctrl+C)...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatıldı.")

if __name__ == '__main__':
    run_server()
