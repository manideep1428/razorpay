"""Minimal, pure HTML web server for testing FraudShield AI models from Hugging Face.

Zero CSS, Zero JavaScript, Zero web framework dependencies (uses Python built-in http.server).
Loads models directly from Hugging Face Hub: vicky1428/fraudshield-models (or local cache).
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import socket
import sys
from urllib.parse import parse_qs

# Force IPv4 socket resolution for Windows/Colab network reliability
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res
socket.getaddrinfo = _ipv4_getaddrinfo

# Ensure src/ is in sys.path
_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from trust_radar.config import FeatureConfig
from trust_radar.inference.predict_payment import PaymentPredictor
from trust_radar.inference.predict_signup import SignupPredictor
from trust_radar.utils.preprocessing import build_graph_data
from trust_radar.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
    synthesize_signup_edges,
)

HF_REPO_ID = "vicky1428/fraudshield-models"
LOCAL_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

print("=" * 70)
print(f"Loading models from Hugging Face: {HF_REPO_ID}")
print("=" * 70)

# 1. Download or locate Payment Abuse Model
payment_path = LOCAL_ARTIFACTS / "payment_abuse_lgbm.joblib"
if not payment_path.exists():
    print(f"Downloading payment model from https://huggingface.co/{HF_REPO_ID}...")
    payment_path = hf_hub_download(repo_id=HF_REPO_ID, filename="payment_abuse_lgbm.joblib")
print(f"[LOADED] Payment model from: {payment_path}")
payment_predictor = PaymentPredictor(payment_path)

# 2. Download or locate Signup Trust Model
signup_path = LOCAL_ARTIFACTS / "signup_graphsage.pt"
if not signup_path.exists():
    print(f"Downloading signup GNN model from https://huggingface.co/{HF_REPO_ID}...")
    signup_path = hf_hub_download(repo_id=HF_REPO_ID, filename="signup_graphsage.pt")
print(f"[LOADED] Signup GNN model from: {signup_path}")
signup_predictor = SignupPredictor(signup_path, device=torch.device("cpu"))

feat_cfg = FeatureConfig()


def build_home_html(signup_result=None, payment_result=None):
    signup_html = ""
    if signup_result:
        signup_html = f"""
        <hr>
        <h2>Signup Model Prediction Result</h2>
        <table border="1">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Trust Score (0-100)</td><td><b>{signup_result['trust_score']}</b></td></tr>
            <tr><td>Risk Score (0-100)</td><td>{signup_result['risk_score']}</td></tr>
            <tr><td>Abuse Probability</td><td>{signup_result['abuse_probability']:.4f}</td></tr>
            <tr><td>Risk Level</td><td>{signup_result['risk_level']}</td></tr>
            <tr><td>Signup Action Decision</td><td><b>{signup_result['decision']}</b></td></tr>
        </table>
        <hr>
        """

    payment_html = ""
    if payment_result:
        payment_html = f"""
        <hr>
        <h2>Payment Abuse Prediction Result</h2>
        <table border="1">
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Payment Risk Score (0-100)</td><td><b>{payment_result['payment_risk_score']}</b></td></tr>
            <tr><td>Abuse Type</td><td><b>{payment_result['abuse_type']}</b></td></tr>
            <tr><td>Risk Level</td><td>{payment_result['risk_level']}</td></tr>
            <tr><td>Payment Action Decision</td><td><b>{payment_result['decision']}</b></td></tr>
            <tr><td>Block Target</td><td>{payment_result.get('block_target') or 'None'}</td></tr>
            <tr><td>Model Scored</td><td>{payment_result.get('scored')}</td></tr>
        </table>
        <hr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>FraudShield AI - Plain HTML Model Tester</title>
</head>
<body>
    <h1>FraudShield AI - Model Inference Tester</h1>
    <p>Models loaded from Hugging Face: <a href="https://huggingface.co/{HF_REPO_ID}">https://huggingface.co/{HF_REPO_ID}</a></p>

    {signup_html}
    {payment_html}

    <h2>1. Test Signup Trust Model (GraphSAGE GNN)</h2>
    <form method="POST" action="/test_signup">
        <table border="1">
            <tr>
                <td><label for="device_trust_score">Device Trust Score (0-100):</label></td>
                <td><input type="number" id="device_trust_score" name="device_trust_score" value="85" min="0" max="100"></td>
            </tr>
            <tr>
                <td><label for="device_age_days">Device Age (Days):</label></td>
                <td><input type="number" id="device_age_days" name="device_age_days" value="120" min="0"></td>
            </tr>
            <tr>
                <td><label for="accounts_per_device_7d">Accounts on Device (Last 7 Days):</label></td>
                <td><input type="number" id="accounts_per_device_7d" name="accounts_per_device_7d" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="shared_device_count">Shared Device Count:</label></td>
                <td><input type="number" id="shared_device_count" name="shared_device_count" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="vpn_flag">VPN Detected:</label></td>
                <td>
                    <select id="vpn_flag" name="vpn_flag">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="proxy_flag">Proxy Detected:</label></td>
                <td>
                    <select id="proxy_flag" name="proxy_flag">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="tor_flag">Tor Detected:</label></td>
                <td>
                    <select id="tor_flag" name="tor_flag">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="accounts_per_ip_7d">Accounts on IP (Last 7 Days):</label></td>
                <td><input type="number" id="accounts_per_ip_7d" name="accounts_per_ip_7d" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="session_duration_seconds">Session Duration (Seconds):</label></td>
                <td><input type="number" id="session_duration_seconds" name="session_duration_seconds" value="45" min="0"></td>
            </tr>
        </table>
        <br>
        <button type="submit">Submit Signup Assessment</button>
    </form>

    <hr>

    <h2>2. Test Payment Abuse Model (Multi-Class LightGBM)</h2>
    <form method="POST" action="/test_payment">
        <table border="1">
            <tr>
                <td><label for="plan_type">Plan Type:</label></td>
                <td>
                    <select id="plan_type" name="plan_type">
                        <option value="trial">Trial (scored)</option>
                        <option value="discounted">Discounted (scored)</option>
                        <option value="standard">Standard (full price - auto allow)</option>
                        <option value="enterprise">Enterprise (full price - auto allow)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="trust_score">Upstream Signup Trust Score (0-100):</label></td>
                <td><input type="number" id="trust_score" name="trust_score" value="85" min="0" max="100"></td>
            </tr>
            <tr>
                <td><label for="amount">Payment Amount ($):</label></td>
                <td><input type="number" step="0.01" id="amount" name="amount" value="0.00"></td>
            </tr>
            <tr>
                <td><label for="is_trial">Is Trial:</label></td>
                <td>
                    <select id="is_trial" name="is_trial">
                        <option value="1">Yes (1)</option>
                        <option value="0">No (0)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="is_discounted">Is Discounted:</label></td>
                <td>
                    <select id="is_discounted" name="is_discounted">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="users_per_card_7d">Users Per Card (7 Days):</label></td>
                <td><input type="number" id="users_per_card_7d" name="users_per_card_7d" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="payments_last_5m">Payments Last 5 Minutes:</label></td>
                <td><input type="number" id="payments_last_5m" name="payments_last_5m" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="payments_last_1h">Payments Last 1 Hour:</label></td>
                <td><input type="number" id="payments_last_1h" name="payments_last_1h" value="1" min="0"></td>
            </tr>
            <tr>
                <td><label for="vpn_flag_pay">VPN Flag:</label></td>
                <td>
                    <select id="vpn_flag_pay" name="vpn_flag">
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </td>
            </tr>
            <tr>
                <td><label for="card_bin_risk_score">Card BIN Risk Score (0-100):</label></td>
                <td><input type="number" id="card_bin_risk_score" name="card_bin_risk_score" value="15" min="0" max="100"></td>
            </tr>
        </table>
        <br>
        <button type="submit">Submit Payment Assessment</button>
    </form>
</body>
</html>
"""


class SimpleModelServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        html = build_home_html()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(post_data)

        def get_val(name, default):
            v = form_data.get(name, [default])[0]
            try:
                return float(v)
            except ValueError:
                return v

        if self.path == "/test_signup":
            sample_df = synthesize_signup_dataset(n=10, seed=42)
            # Update target row with submitted form values
            sample_df.loc[0, "device_trust_score"] = get_val("device_trust_score", 85.0)
            sample_df.loc[0, "device_age_days"] = get_val("device_age_days", 120.0)
            sample_df.loc[0, "accounts_per_device_7d"] = get_val("accounts_per_device_7d", 1.0)
            sample_df.loc[0, "shared_device_count"] = get_val("shared_device_count", 1.0)
            sample_df.loc[0, "vpn_flag"] = int(get_val("vpn_flag", 0))
            sample_df.loc[0, "proxy_flag"] = int(get_val("proxy_flag", 0))
            sample_df.loc[0, "tor_flag"] = int(get_val("tor_flag", 0))
            sample_df.loc[0, "accounts_per_ip_7d"] = get_val("accounts_per_ip_7d", 1.0)
            sample_df.loc[0, "session_duration_seconds"] = get_val("session_duration_seconds", 45.0)

            nodes = sample_df[feat_cfg.signup_numeric_features]
            edges = synthesize_signup_edges(len(sample_df), avg_degree=4.0, seed=42)
            graph = build_graph_data(nodes, edges, labels=sample_df["label"])
            res = signup_predictor.predict_single_node(node_idx=0, data=graph)

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = build_home_html(signup_result=res)
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/test_payment":
            sample_df = synthesize_payment_dataset(n=5, seed=42)
            row = sample_df.iloc[[0]].copy()
            plan_type = str(form_data.get("plan_type", ["trial"])[0])
            row["plan_type"] = plan_type
            row["trust_score"] = get_val("trust_score", 85.0)
            row["amount"] = get_val("amount", 0.0)
            row["is_trial"] = int(get_val("is_trial", 1))
            row["is_discounted"] = int(get_val("is_discounted", 0))
            row["users_per_card_7d"] = get_val("users_per_card_7d", 1.0)
            row["payments_last_5m"] = get_val("payments_last_5m", 1.0)
            row["payments_last_1h"] = get_val("payments_last_1h", 1.0)
            row["vpn_flag"] = int(get_val("vpn_flag", 0))
            row["card_bin_risk_score"] = get_val("card_bin_risk_score", 15.0)

            res = payment_predictor.score_transaction(row, plan_type=plan_type)

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = build_home_html(payment_result=res)
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, SimpleModelServer)
    print(f"\n[SERVER RUNNING] Open in browser: http://localhost:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
