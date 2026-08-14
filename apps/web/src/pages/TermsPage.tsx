// 컴플라이언스 핸드오프(complience/Compliance & Securit.md, 2026-08-14)의
// 이용약관 초안을 프로젝트 실제 동작에 맞게 다듬어 반영. 원본 초안은 법률
// 검토를 거치지 않은 팀원 작업물이라는 점을 그대로 유지 -- 이 페이지 역시
// 정식 법률 자문을 대체하지 않는 안내문임을 명시한다.
export function TermsPage() {
  return (
    <div className="mx-auto mt-8 max-w-2xl pb-16 text-sm leading-relaxed text-neutral-300">
      <h1 className="mb-1 text-2xl font-semibold text-neutral-100">이용약관 및 AI 학습 방지 정책</h1>
      <p className="mb-8 text-xs text-neutral-500">
        본 문서는 정식 법률 자문을 거치지 않은 초안이며, 서비스의 실제 동작 방식을 설명하기 위한 안내 목적입니다.
      </p>

      <section className="mb-8">
        <h2 className="mb-2 text-base font-medium text-neutral-100">제 1 조: 창작물의 보호 및 AI 학습 금지</h2>
        <ol className="list-decimal space-y-2 pl-5">
          <li>본 플랫폼에 게시된 모든 디지털 창작물의 저작권은 원작자에게 있습니다.</li>
          <li>
            창작자는 업로드 시 AI 학습 허용 여부를 선택할 수 있으며, 이 선택은 온체인에 기록되는 doNotTrain
            플래그와 C2PA 메타데이터에 반영됩니다. 이는 법적·선언적 신호로, 별도로 적용되는 기술적 보호조치(아래 제
            2 조)와는 독립적으로 작동합니다.
          </li>
          <li>
            이 조항을 위반한 무단 학습·도용이 확인될 경우, 플랫폼은 온체인 타임스탬프와 실제 서명 키(Ed25519,
            KMS 관리) 기반 증거 패키지를 생성해 민·형사상 조치를 지원할 수 있습니다. 증거 패키지는 테스트
            랩(Test Lab)의 도용 탐지 케이스에서 직접 확인·다운로드할 수 있습니다.
          </li>
        </ol>
      </section>

      <section className="mb-8">
        <h2 className="mb-2 text-base font-medium text-neutral-100">제 2 조: 기술적 보호조치 및 서비스 면책</h2>
        <ol className="list-decimal space-y-2 pl-5">
          <li>
            플랫폼이 제공하는 AI 학습 방지 기능(L1~L3, STRONG_PROTECTION)은 기술적 억제 수단이며, 어떠한 보호
            등급도 AI 학습의 100% 차단을 보증하지 않습니다.
          </li>
          <li>
            보호 강도가 높을수록 원본 대비 화질 저하가 발생할 수 있습니다. 이는 학습 방해를 위한 의도적 가공이며,
            플랫폼은 이로 인한 손해에 대해 책임을 지지 않습니다.
          </li>
          <li>
            플랫폼이 발급하는 증거 패키지는 기술적·정황적 무결성을 증명하는 자료이며, 플랫폼이 직접 법적 대리인
            으로서 소송을 대행하거나 결과를 보장하지 않습니다.
          </li>
        </ol>
      </section>
    </div>
  );
}
