import type { ReactNode } from "react";

/** 목록과 상세 화면에서 공통으로 쓰는 로딩·오류·빈 결과 표시다. */
export function LoadingState({ label = "데이터를 불러오는 중입니다." }: { label?: string }) {
  return <p className="state-message" role="status">{label}</p>;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="state-message state-message--empty">{children}</div>;
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="state-message state-message--error" role="alert">
      <p>데이터를 불러오지 못했습니다.</p>
      <p className="state-message__detail">{error.message}</p>
      <button type="button" onClick={onRetry}>다시 시도</button>
    </div>
  );
}
