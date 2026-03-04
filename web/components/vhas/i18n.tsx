"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type Locale = "en" | "vi";

interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");

  useEffect(() => {
    const stored = window.localStorage.getItem("vhas-locale") as Locale | null;
    if (stored === "en" || stored === "vi") {
      setLocale(stored);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("vhas-locale", locale);
  }, [locale]);

  const value = useMemo(() => ({ locale, setLocale }), [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return context;
}

export function LanguageToggle() {
  const { locale, setLocale } = useI18n();

  return (
    <div className="inline-flex items-center rounded-full border border-slate-200 bg-white/80 p-1">
      {(["en", "vi"] as const).map((value) => (
        <Button
          key={value}
          size="sm"
          variant="ghost"
          className={cn(
            "h-8 rounded-full px-3 text-xs font-semibold",
            locale === value
              ? "bg-blue-600 text-white hover:bg-blue-600"
              : "text-slate-600 hover:bg-slate-100"
          )}
          onClick={() => setLocale(value)}
        >
          {value === "en" ? "EN" : "VI"}
        </Button>
      ))}
    </div>
  );
}
