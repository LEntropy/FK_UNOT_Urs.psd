import { Wallet } from "ethers";
import { wrapKey } from "@dontai/kms-adapter";
import { env } from "../env.js";

export interface ProvisionedWallet {
  address: string;
  encryptedPrivateKeyBase64: string;
}

/**
 * Platform custodial wallet (PROJECT_DESIGN.md §3-1: "초기엔 플랫폼 커스터디
 * 지갑, 서버가 대리 서명, KMS로 키 보관"). The private key never touches
 * asset-service or the frontend -- only the RSA-wrapped ciphertext is
 * persisted (users.encryptedWalletKey). Unwrapping it back (via
 * infra/kms-adapter's unwrapKey against the live KMS server) is left for
 * whichever future flow actually needs the platform to sign a transaction
 * on the user's behalf; nothing in this pass does that yet.
 */
export function provisionCustodialWallet(): ProvisionedWallet {
  const wallet = Wallet.createRandom();
  const privateKeyBytes = Buffer.from(wallet.privateKey.slice(2), "hex"); // strip 0x
  const wrapped = wrapKey(env.KMS_PUBLIC_KEY_PATH, privateKeyBytes);

  return {
    address: wallet.address,
    encryptedPrivateKeyBase64: wrapped.toString("base64"),
  };
}
