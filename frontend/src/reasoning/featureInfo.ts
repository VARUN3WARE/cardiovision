import { useEffect, useState } from "react";
import { getFeatures, type FeaturesResponse } from "../api/client";

let request: Promise<FeaturesResponse | null> | null = null;

/**
 * /features (schema keys to column names, calculated-score inputs): the caller's copy when it has one, else
 * fetched once while `enabled` (a failed request is retried when `retryKey` changes, e.g. on a new result).
 */
export function useFeatureInfo(given: FeaturesResponse | null | undefined, enabled: boolean, retryKey: unknown) {
  const [fetched, setFetched] = useState<FeaturesResponse | null>(null);
  const need = enabled && !given && !fetched;
  useEffect(() => {
    if (!need) return;
    let live = true;
    request ??= getFeatures().catch(() => {
      request = null;
      return null;
    });
    request.then((f) => {
      if (live && f) setFetched(f);
    });
    return () => {
      live = false;
    };
  }, [need, retryKey]);
  return given ?? fetched;
}
