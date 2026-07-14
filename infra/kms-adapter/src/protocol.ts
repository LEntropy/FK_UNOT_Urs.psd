// Wire-format mirror of the C KMS server's src/protocol.c.
//
// Request (all ints big-endian int32, strings length-prefixed UTF-8):
//   requesterOrgLen + requesterOrg
//   fileOrgLen      + fileOrg
//   keyIdLen        + keyId
//   encKeyLen       + encKey (bytes)
//
// Response:
//   resultCode (int32) -- 0 success, -403 access denied (org policy),
//     -451 expired / -452 revoked / -453 unknown key / -460 policy load
//     failure, -2..-99 other server errors
//   if resultCode === 0: plainKeyLen (int32) + plainKey (bytes)

import type { Socket } from "node:net";

export function encodeUnwrapRequest(requesterOrg: string, fileOrg: string, keyId: string, encKey: Buffer): Buffer {
  const parts = [requesterOrg, fileOrg, keyId].map((s) => Buffer.from(s, "utf8"));
  const chunks: Buffer[] = [];
  for (const part of parts) {
    const len = Buffer.alloc(4);
    len.writeInt32BE(part.length, 0);
    chunks.push(len, part);
  }
  const encLen = Buffer.alloc(4);
  encLen.writeInt32BE(encKey.length, 0);
  chunks.push(encLen, encKey);
  return Buffer.concat(chunks);
}

function readExact(socket: Socket, length: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const buf = Buffer.alloc(length);
    let offset = 0;

    const cleanup = () => {
      socket.off("readable", onReadable);
      socket.off("error", onError);
      socket.off("close", onClose);
    };
    const onError = (err: Error) => {
      cleanup();
      reject(err);
    };
    const onClose = () => {
      cleanup();
      reject(new Error(`KMS connection closed after ${offset}/${length} bytes`));
    };
    const onReadable = () => {
      let chunk: Buffer | null;
      while (offset < length && (chunk = socket.read(length - offset)) !== null) {
        chunk.copy(buf, offset);
        offset += chunk.length;
      }
      if (offset >= length) {
        cleanup();
        resolve(buf);
      }
    };

    socket.on("readable", onReadable);
    socket.on("error", onError);
    socket.on("close", onClose);
    onReadable();
  });
}

export async function readInt32BE(socket: Socket): Promise<number> {
  const buf = await readExact(socket, 4);
  return buf.readInt32BE(0);
}

export async function readUnwrapResponse(socket: Socket): Promise<Buffer> {
  const resultCode = await readInt32BE(socket);
  if (resultCode !== 0) {
    throw new KmsProtocolError(resultCode);
  }
  const plainKeyLen = await readInt32BE(socket);
  return readExact(socket, plainKeyLen);
}

const RESULT_CODE_MESSAGES: Record<number, string> = {
  [-403]: "access denied (requesterOrg cannot decrypt fileOrg)",
  [-451]: "key expired",
  [-452]: "key revoked",
  [-453]: "unknown key (not in policy.conf)",
  [-460]: "KMS policy file failed to load",
};

export class KmsProtocolError extends Error {
  constructor(public readonly resultCode: number) {
    super(RESULT_CODE_MESSAGES[resultCode] ?? `KMS server error (resultCode=${resultCode})`);
    this.name = "KmsProtocolError";
  }
}
