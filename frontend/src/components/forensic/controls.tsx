import { cva, type VariantProps } from "class-variance-authority";
import { motion } from "framer-motion";
import { Download, Play, UploadCloud } from "lucide-react";
import { useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/utils";

const actionButton = cva(
  "inline-flex items-center justify-center gap-2 rounded-xl text-sm font-medium transition-all focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        action:
          "gradient-action text-primary-foreground shadow-[0_10px_30px_-12px_var(--primary)] hover:-translate-y-0.5 hover:shadow-[0_14px_36px_-10px_var(--primary)]",
        glass:
          "border border-border bg-glass text-foreground backdrop-blur-md hover:border-primary-bright/40 hover:bg-accent/40",
        ghost: "text-muted-foreground hover:bg-accent/40 hover:text-foreground",
        danger: "border border-destructive/30 bg-destructive/10 text-destructive hover:bg-destructive/20",
      },
      size: {
        sm: "h-9 px-3",
        md: "h-11 px-5",
        lg: "h-12 px-6 text-base",
        icon: "size-10",
      },
    },
    defaultVariants: { variant: "glass", size: "md" },
  },
);

export type ActionButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof actionButton>;

export function ActionButton({ className, variant, size, ...props }: ActionButtonProps) {
  return <button className={cn(actionButton({ variant, size }), className)} {...props} />;
}

export function DropZone({
  label,
  hint,
  accept,
  multiple = true,
  onFiles,
}: {
  label: string;
  hint: string;
  accept: string;
  multiple?: boolean;
  onFiles: (files: File[]) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={label}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFiles(Array.from(e.dataTransfer.files));
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center gap-2 rounded-2xl border border-dashed px-6 py-10 text-center transition-all focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
        dragging
          ? "border-primary-bright bg-primary/10"
          : "border-border bg-muted/10 hover:border-primary-bright/50 hover:bg-primary/5",
      )}
    >
      <span className="rounded-2xl border border-border bg-primary/10 p-3 text-primary-bright">
        <UploadCloud className="size-6" />
      </span>
      <p className="text-sm font-medium text-foreground">{label}</p>
      <p className="max-w-xs text-xs text-muted-foreground">{hint}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="sr-only"
        onChange={(e) => onFiles(Array.from(e.target.files ?? []))}
      />
    </div>
  );
}

/**
 * Renders a real screenshot image or annotated video when the backend URL is
 * provided, falling back to a gradient placeholder for pre-API mocks.
 */
export function VideoSurface({
  gradient,
  src,
  videoSrc,
  seekTo,
  caption,
  overlay,
  onPlay,
  aspect = "aspect-video",
}: {
  gradient: string;
  /** Image URL of a match screenshot. */
  src?: string | undefined;
  /** Video URL of an annotated clip or the highlight reel. */
  videoSrc?: string | undefined;
  /** Seek position (seconds) for video playback. */
  seekTo?: number;
  caption?: string;
  overlay?: ReactNode;
  onPlay?: () => void;
  aspect?: string;
}) {
  return (
    <div className={cn("relative w-full overflow-hidden rounded-xl border border-border", aspect)}>
      {videoSrc ? (
        <video
          key={seekTo !== undefined ? `${videoSrc}#${seekTo}` : videoSrc}
          controls
          preload="metadata"
          muted
          playsInline
          className="absolute inset-0 size-full rounded-xl object-contain bg-black"
        >
          <source
            src={seekTo !== undefined ? `${videoSrc}#t=${seekTo}` : videoSrc}
            type="video/mp4"
          />
        </video>
      ) : src ? (
        <img src={src} alt={caption ?? "Screenshot"} className="absolute inset-0 size-full object-cover" />
      ) : (
        <>
          <div className={cn("absolute inset-0 bg-gradient-to-br", gradient)} />
          <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_3px,rgba(0,0,0,0.18)_4px)] opacity-40" />
        </>
      )}
      {overlay}
      {!videoSrc ? (
        <button
          type="button"
          onClick={onPlay}
          aria-label={caption ? `Play ${caption}` : "Play"}
          className="absolute inset-0 flex items-center justify-center focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <motion.span
            whileHover={{ scale: 1.08 }}
            className="glass-panel flex size-12 items-center justify-center rounded-full p-0 text-primary-bright"
          >
            <Play className="size-5" />
          </motion.span>
        </button>
      ) : null}
      {caption ? (
        <span className="absolute right-2 bottom-2 left-2 z-10 rounded-md bg-background/70 px-2 py-1 text-center font-mono text-[11px] text-foreground backdrop-blur">
          {caption}
        </span>
      ) : null}
    </div>
  );
}

export function DownloadButton({
  label,
  onClick,
  pending,
}: {
  label: string;
  onClick: () => void;
  pending?: boolean;
}) {
  return (
    <ActionButton variant="glass" size="sm" onClick={onClick} disabled={pending}>
      <Download className={cn("size-4", pending && "animate-pulse")} />
      {pending ? "Preparing…" : label}
    </ActionButton>
  );
}
