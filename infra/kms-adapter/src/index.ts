import { readFileSync } from "node:fs";
import { publicEncrypt, constants as cryptoConstants } from "node:crypto";
import { connect as tlsConnect, type ConnectionOptions } from "node:tls";
import { encodeUnwrapRequest, readUnwrapResponse } from "./protocol.js";

export { KmsProtocolError } from "./protocol.js";

/**
 * Envelope-encrypts a plaintext key (e.g. a per-artwork DEK) against an org's
 * RSA public key. Purely client-side -- the KMS server only ever decrypts
 * (see unwrapKey), never wraps, so this never touches the network.
 *
 * Padding MUST be PKCS#1 v1.5 (RSA_PKCS1_PADDING), matching the server's
 * RSA_private_decrypt call in protocol.c -- OAEP here would fail to decrypt
 * on the server side.
 */
export function wrapKey(publicKeyPemPath: string, plainKey: Buffer): Buffer {
  const publicKey = readFileSync(publicKeyPemPath, "utf8");
  return publicEncrypt(
    { key: publicKey, padding: cryptoConstants.RSA_PKCS1_PADDING },
    plainKey,
  );
}

export interface UnwrapKeyOptions {
  host: string;
  port: number;
  /** CA certificate used to verify the KMS server's TLS cert (pinned, not a public CA). */
  caCertPath: string;
  /** Org path the caller is authenticating as, e.g. "teamA" or "teamA/teamA1". */
  requesterOrg: string;
  /** Org path the encrypted key belongs to (the .enc file's namespace). */
  fileOrg: string;
  keyId: string;
  encKey: Buffer;
  timeoutMs?: number;
}

/**
 * Calls the live KMS server to decrypt an envelope-encrypted key.
 * The KMS server is internal-only (never exposed publicly, per
 * PROJECT_DESIGN.md §6-2) and its cert is pinned by caCertPath rather than
 * validated against a public CA or hostname, so we skip hostname/SAN
 * checking and rely purely on chain-of-trust to the pinned CA.
 */
export function unwrapKey(opts: UnwrapKeyOptions): Promise<Buffer> {
  const { host, port, caCertPath, requesterOrg, fileOrg, keyId, encKey, timeoutMs = 10_000 } = opts;

  const tlsOptions: ConnectionOptions = {
    host,
    port,
    ca: readFileSync(caCertPath),
    rejectUnauthorized: true,
    checkServerIdentity: () => undefined, // pinned by CA, not by hostname
  };

  return new Promise((resolve, reject) => {
    const socket = tlsConnect(tlsOptions, () => {
      socket.write(encodeUnwrapRequest(requesterOrg, fileOrg, keyId, encKey));
    });

    socket.setTimeout(timeoutMs, () => {
      socket.destroy(new Error(`KMS request timed out after ${timeoutMs}ms`));
    });

    socket.once("error", reject);

    readUnwrapResponse(socket)
      .then((plainKey) => {
        socket.end();
        resolve(plainKey);
      })
      .catch((err) => {
        socket.destroy();
        reject(err);
      });
  });
}
