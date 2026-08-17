import { Routes, Route, Navigate } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { RightSidebar } from "./components/RightSidebar";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { SignupPage } from "./pages/SignupPage";
import { UploadPage } from "./pages/UploadPage";
import { GalleryPage } from "./pages/GalleryPage";
import { FeedPage } from "./pages/FeedPage";
import { ModerationPage } from "./pages/ModerationPage";
import { ArtworkDetailPage } from "./pages/ArtworkDetailPage";
import { OAuthCallbackPage } from "./pages/OAuthCallbackPage";
import { TestLabPage } from "./pages/TestLabPage";
import { TermsPage } from "./pages/TermsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ProfilePage } from "./pages/ProfilePage";
import { FollowListPage } from "./pages/FollowListPage";
import { useAuthStore } from "./store/auth";

export function App() {
  const isAuthed = useAuthStore((s) => !!s.user);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <NavBar />
      <main className={isAuthed ? "px-4 pb-20 pt-4 md:ml-20 md:px-8 md:pb-8 xl:ml-64 xl:mr-80" : "px-6 py-4"}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/oauth-callback" element={<OAuthCallbackPage />} />
          <Route path="/terms" element={<TermsPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/" element={<FeedPage />} />
            <Route path="/my-artworks" element={<GalleryPage />} />
            <Route path="/moderation" element={<ModerationPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/test-lab" element={<TestLabPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/artworks/:id" element={<ArtworkDetailPage />} />
            <Route path="/creators/:id" element={<ProfilePage />} />
            <Route path="/creators/:id/followers" element={<FollowListPage kind="followers" />} />
            <Route path="/creators/:id/following" element={<FollowListPage kind="following" />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      {isAuthed && (
        <aside className="fixed inset-y-0 right-0 hidden w-80 overflow-y-auto px-4 py-6 xl:block">
          <RightSidebar />
        </aside>
      )}
    </div>
  );
}
