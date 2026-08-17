import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/clients/assetService.js", () => ({
  createArtwork: vi.fn(),
  createArtworkWithFile: vi.fn(),
  listArtworks: vi.fn(),
  getArtwork: vi.fn(),
  suggestTags: vi.fn(),
  AssetServiceError: class AssetServiceError extends Error {
    constructor(public status: number, public body: unknown) {
      super("asset-service request failed");
    }
  },
}));
vi.mock("../src/clients/deliveryGateway.js", () => ({ signRenderUrl: vi.fn() }));
vi.mock("../src/clients/detectionService.js", () => ({
  scanArtwork: vi.fn(),
  reportArtwork: vi.fn(),
  reportModelLeak: vi.fn(),
  getCase: vi.fn(),
  getEvidence: vi.fn(),
  getDmcaNotice: vi.fn(),
  DetectionServiceError: class DetectionServiceError extends Error {
    constructor(public status: number, public body: unknown) {
      super("detection-svc request failed");
    }
  },
}));

const { getArtwork, AssetServiceError } = await import("../src/clients/assetService.js");
const { scanArtwork, reportArtwork, reportModelLeak, getCase, getEvidence, getDmcaNotice, DetectionServiceError } = await import(
  "../src/clients/detectionService.js"
);

beforeEach(() => {
  vi.clearAllMocks();
});

async function signupAndGetToken(app: ReturnType<typeof createApp>) {
  const res = await request(app)
    .post("/auth/signup")
    .send({ email: "detect@example.com", password: "hunter22", handle: "detect_user" });
  return { accessToken: res.body.accessToken as string, userId: res.body.user.id as string };
}

describe("POST /artworks/:id/scan", () => {
  it("rejects unauthenticated requests", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).post("/artworks/ast_1/scan");
    expect(res.status).toBe(401);
    expect(scanArtwork).not.toHaveBeenCalled();
  });

  it("403s when the caller isn't this artwork's creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app).post("/artworks/ast_1/scan").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(403);
    expect(scanArtwork).not.toHaveBeenCalled();
  });

  it("triggers a real scan for the artwork's own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });
    vi.mocked(scanArtwork).mockResolvedValue({ caseId: "case_abc", status: "queued" });

    const res = await request(app).post("/artworks/ast_1/scan").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(202);
    expect(res.body).toEqual({ caseId: "case_abc", status: "queued" });
    expect(scanArtwork).toHaveBeenCalledWith("ast_1");
  });

  it("404s when the artwork doesn't exist", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockRejectedValue(new AssetServiceError(404, { error: "no artwork ast_missing" }));

    const res = await request(app).post("/artworks/ast_missing/scan").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(404);
  });
});

describe("POST /artworks/:id/report", () => {
  it("400s on a missing/invalid suspectUrl", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    const res = await request(app)
      .post("/artworks/ast_1/report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectUrl: "not-a-url" });
    expect(res.status).toBe(400);
    expect(getArtwork).not.toHaveBeenCalled();
  });

  it("403s when the caller isn't this artwork's creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app)
      .post("/artworks/ast_1/report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectUrl: "https://example.com/found.png" });
    expect(res.status).toBe(403);
    expect(reportArtwork).not.toHaveBeenCalled();
  });

  it("forwards a real report for the artwork's own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });
    vi.mocked(reportArtwork).mockResolvedValue({ caseId: "case_def", status: "queued" });

    const res = await request(app)
      .post("/artworks/ast_1/report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectUrl: "https://example.com/found.png" });
    expect(res.status).toBe(202);
    expect(reportArtwork).toHaveBeenCalledWith("ast_1", "https://example.com/found.png");
  });
});

describe("POST /artworks/:id/model-leak-report", () => {
  it("400s on a missing/invalid suspectModelUrl", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    const res = await request(app)
      .post("/artworks/ast_1/model-leak-report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectModelUrl: "not-a-url" });
    expect(res.status).toBe(400);
    expect(getArtwork).not.toHaveBeenCalled();
  });

  it("403s when the caller isn't this artwork's creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app)
      .post("/artworks/ast_1/model-leak-report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectModelUrl: "https://civitai.com/models/12345" });
    expect(res.status).toBe(403);
    expect(reportModelLeak).not.toHaveBeenCalled();
  });

  it("forwards a real model-leak report for the artwork's own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });
    vi.mocked(reportModelLeak).mockResolvedValue({ caseId: "case_leak1", status: "queued" });

    const res = await request(app)
      .post("/artworks/ast_1/model-leak-report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectModelUrl: "https://civitai.com/models/12345" });
    expect(res.status).toBe(202);
    expect(reportModelLeak).toHaveBeenCalledWith("ast_1", "https://civitai.com/models/12345");
  });

  it("404s when the artwork doesn't exist", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getArtwork).mockRejectedValue(new AssetServiceError(404, { error: "no artwork ast_missing" }));

    const res = await request(app)
      .post("/artworks/ast_missing/model-leak-report")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ suspectModelUrl: "https://civitai.com/models/12345" });
    expect(res.status).toBe(404);
  });
});

describe("GET /detection-cases/:caseId", () => {
  it("rejects unauthenticated requests", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).get("/detection-cases/case_1");
    expect(res.status).toBe(401);
  });

  it("403s when the case's artwork isn't the caller's own", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getCase).mockResolvedValue({
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY",
      trigger: "report",
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    });
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app).get("/detection-cases/case_1").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(403);
  });

  it("returns the case for its own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    const detectionCase = {
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY" as const,
      trigger: "report" as const,
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    };
    vi.mocked(getCase).mockResolvedValue(detectionCase);
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });

    const res = await request(app).get("/detection-cases/case_1").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(res.body).toEqual(detectionCase);
  });

  it("404s when the case doesn't exist", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getCase).mockRejectedValue(new DetectionServiceError(404, { error: "no case case_missing" }));

    const res = await request(app).get("/detection-cases/case_missing").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(404);
  });
});

describe("GET /detection-cases/:caseId/evidence", () => {
  it("403s when the case's artwork isn't the caller's own", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getCase).mockResolvedValue({
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY",
      trigger: "report",
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    });
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app)
      .get("/detection-cases/case_1/evidence")
      .set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(403);
    expect(getEvidence).not.toHaveBeenCalled();
  });

  it("returns the evidence bundles for the case's own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(getCase).mockResolvedValue({
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY",
      trigger: "report",
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    });
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });
    vi.mocked(getEvidence).mockResolvedValue({ caseId: "case_1", status: "EVIDENCE_READY", bundles: [{ discoveredUrl: "https://x" } as never] });

    const res = await request(app)
      .get("/detection-cases/case_1/evidence")
      .set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(res.body.bundles).toHaveLength(1);
  });
});

describe("GET /detection-cases/:caseId/dmca-notice", () => {
  it("403s when the case's artwork isn't the caller's own", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(getCase).mockResolvedValue({
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY",
      trigger: "report",
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    });
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: "someone-else" });

    const res = await request(app)
      .get("/detection-cases/case_1/dmca-notice")
      .set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(403);
    expect(getDmcaNotice).not.toHaveBeenCalled();
  });

  it("returns the auto-filled notice for the case's own creator", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(getCase).mockResolvedValue({
      id: "case_1",
      artwork_id: "ast_1",
      status: "EVIDENCE_READY",
      trigger: "report",
      error_message: null,
      note: null,
      created_at: 0,
      updated_at: 0,
      evidence: [],
    });
    vi.mocked(getArtwork).mockResolvedValue({ id: "ast_1", creatorId: userId });
    vi.mocked(getDmcaNotice).mockResolvedValue({
      caseId: "case_1",
      notices: [{ sourceUrl: "https://x", notice: "To: [host]...", note: null }],
    });

    const res = await request(app)
      .get("/detection-cases/case_1/dmca-notice")
      .set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(res.body.notices).toHaveLength(1);
    expect(res.body.notices[0].notice).toContain("To: [host]");
  });
});
