/** Backend의 공통 오류 계약을 TypeScript 오류 객체로 바꾸는 HTTP client다. */

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type ErrorResponse = {
  error?: {
    code?: string;
    message?: string;
  };
};

/** API 기본 경로를 한 곳에 두어 배포 환경에서만 환경 변수로 바꿀 수 있게 한다. */
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

/** JSON 성공 응답과 구조화된 오류 응답을 일관되게 처리한다. */
export async function getJson<Response>(
  path: string,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: { Accept: "application/json" },
    signal,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as ErrorResponse;
    throw new ApiError(
      response.status,
      payload.error?.code ?? "api_request_failed",
      payload.error?.message ?? `요청에 실패했습니다. (HTTP ${response.status})`,
    );
  }

  return (await response.json()) as Response;
}

/** 변경 요청도 조회와 같은 구조화된 오류 계약으로 처리한다. */
export async function postJson<Response>(path: string, payload?: unknown): Promise<Response> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => ({}))) as ErrorResponse;
    throw new ApiError(response.status, errorPayload.error?.code ?? "api_request_failed", errorPayload.error?.message ?? `요청에 실패했습니다. (HTTP ${response.status})`);
  }
  return (await response.json()) as Response;
}
