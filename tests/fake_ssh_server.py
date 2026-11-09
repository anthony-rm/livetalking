"""Minimal in-process SSH server used ONLY to verify remote/ssh_test.py.

Not part of the app: it exists so the SSH login path can be exercised without
touching the machine's real sshd (enabling Remote Login needs sudo).

Run standalone:  python tests/fake_ssh_server.py --port 2222 --password pw
"""
from __future__ import annotations

import argparse
import socket
import threading

import paramiko


class _Server(paramiko.ServerInterface):
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.authed = False

    def check_auth_password(self, username, password):
        if username == self.username and password == self.password:
            self.authed = True
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "password"

    def check_channel_request(self, kind, chanid):
        if kind == "session":
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_exec_request(self, channel, command):
        # Single fixed reply keeps the fake server trivial. Send the exit status
        # and close so the client's stdout.read() sees EOF instead of blocking.
        channel.send(b"FAKE-PC-HOST\n")
        channel.send_exit_status(0)
        channel.shutdown_write()
        return True


def serve(port: int, username: str, password: str, host_key_path: str,
          ready=None):
    key = paramiko.RSAKey(filename=host_key_path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(5)
    if ready:
        ready.set()

    while True:
        try:
            client, _addr = sock.accept()
        except OSError:
            return
        try:
            transport = paramiko.Transport(client)
            transport.add_server_key(key)
            transport.start_server(server=_Server(username, password))
            chan = transport.accept(10)
            if chan is not None:
                try:
                    chan.recv(1024)
                except Exception:  # noqa: BLE001
                    pass
                chan.close()
            transport.close()
        except Exception:  # noqa: BLE001
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2222)
    p.add_argument("--username", default="testuser")
    p.add_argument("--password", default="Sup3rS3cret!pw")
    p.add_argument("--host-key", default="/tmp/fake_ssh_host_rsa")
    a = p.parse_args()
    threading.Thread(target=serve, args=(a.port, a.username, a.password,
                                         a.host_key), daemon=True).start()
    print(f"fake ssh on 127.0.0.1:{a.port}")
    threading.Event().wait()


if __name__ == "__main__":
    main()
