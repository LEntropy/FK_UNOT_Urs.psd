import { createApp } from "./app.js";
import { env } from "./env.js";
import { startBalancePoller } from "./relayerBalance.js";

createApp().listen(env.PORT, () => {
  console.log(`blockchain-svc listening on http://localhost:${env.PORT}`);
});

startBalancePoller(env.RELAYER_BALANCE_POLL_INTERVAL_SECONDS * 1000);
