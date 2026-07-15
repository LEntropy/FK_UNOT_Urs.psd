import { Routes, Route, Navigate } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { SignupPage } from "./pages/SignupPage";
import { UploadPage } from "./pages/UploadPage";
import { GalleryPage } from "./pages/GalleryPage";
import { FeedPage } from "./pages/FeedPage";
import { ModerationPage } from "./pages/ModerationPage";
import { ArtworkDetailPage } from "./pages/ArtworkDetailPage";
import { OAuthCallbackPage } from "./pages/OAuthCallbackPage";

export function App() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <NavBar />
      <main className="px-6 py-4">
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/oauth-callback" element={<OAuthCallbackPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/" element={<FeedPage />} />
            <Route path="/my-artworks" element={<GalleryPage />} />
            <Route path="/moderation" element={<ModerationPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/artworks/:id" element={<ArtworkDetailPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
