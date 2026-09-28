import React from "react";

import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";

export interface PassThroughSecuritySectionProps {
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({ authEnabled, onAuthChange }) => {
  const authId = React.useId();

  return (
    <Card className="block p-6">
      <h3 className="mb-2 text-lg font-semibold text-foreground">Security</h3>
      <p className="mb-4 text-sm text-muted-foreground">
        When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key. When disabled, requests can
        reach the upstream API without LiteLLM key authentication.
      </p>
      <div className="flex items-center gap-2">
        <Switch id={authId} checked={authEnabled} onCheckedChange={onAuthChange} />
        <label htmlFor={authId} className="text-sm text-foreground">
          Require Virtual Key
        </label>
      </div>
    </Card>
  );
};

export default PassThroughSecuritySection;
